## l2_attack.py -- attack a network optimizing for l_2 distance
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import sys
import tensorflow as tf
import numpy as np

BINARY_SEARCH_STEPS = 9  # number of times to adjust the constant with binary search
MAX_ITERATIONS = 10000   # number of iterations to perform gradient descent
ABORT_EARLY = False       # if we stop improving, abort gradient descent early
LEARNING_RATE = 1e-2     # larger values converge faster to less accurate results
TARGETED = False          # should we target one specific class? or just be wrong?
CONFIDENCE = 0           # how strong the adversarial example should be
# INITIAL_CONST = 1e0     # the initial constant c to pick as a first guess

class CarliniL2:
    def __init__(self, sess, model, batch_size=1, self_db_size=1, target_batch_size=1, confidence = CONFIDENCE,
                 targeted = TARGETED, learning_rate = LEARNING_RATE,
                 binary_search_steps = BINARY_SEARCH_STEPS, max_iterations = MAX_ITERATIONS,
                 abort_early = ABORT_EARLY, hinge_loss = True,
                 initial_const = 1e0,
                 boxmin = 0, boxmax = 1):
        """
        The L_2 optimized attack. 

        This attack is the most efficient and should be used as the primary 
        attack to evaluate potential defenses.

        Returns adversarial examples for the supplied model.

        confidence: Confidence of adversarial examples: higher produces examples
          that are farther away, but more strongly classified as adversarial.
        batch_size: Number of attacks to run simultaneously.
        targeted: True if we should perform a targetted attack, False otherwise.
        learning_rate: The learning rate for the attack algorithm. Smaller values
          produce better results but are slower to converge.
        binary_search_steps: The number of times we perform binary search to
          find the optimal tradeoff-constant between distance and confidence. 
        max_iterations: The maximum number of iterations. Larger values are more
          accurate; setting too small will require a large learning rate and will
          produce poor results.
        abort_early: If true, allows early aborts if gradient descent gets stuck.
        initial_const: The initial tradeoff-constant to use to tune the relative
          importance of distance and confidence. If binary_search_steps is large,
          the initial constant is not important.
        boxmin: Minimum pixel value (default -0.5).
        boxmax: Maximum pixel value (default 0.5).
        """
        
        image_height, image_width, num_channels = model.image_height, model.image_width, model.num_channels
        self.sess = sess
        self.TARGETED = targeted
        self.LEARNING_RATE = learning_rate
        self.MAX_ITERATIONS = max_iterations
        self.BINARY_SEARCH_STEPS = binary_search_steps
        self.ABORT_EARLY = abort_early
        self.CONFIDENCE = confidence
        self.initial_const = initial_const
        self.batch_size = batch_size
        self.target_batch_size = target_batch_size
        self.self_db_size = self_db_size
        self.hinge_loss = hinge_loss

        self.repeat = binary_search_steps >= 10

        shape = (batch_size,num_channels,image_height,image_width)
        target_shape = (target_batch_size,num_channels,image_height,image_width)
        self_db_shape = (self_db_size,num_channels,image_height,image_width)
        
        # the variable we're going to optimize over
        modifier = tf.Variable(tf.random_uniform(shape, minval=-0.1, maxval=0.1, dtype=tf.float32))
        

        # these are variables to be more efficient in sending data to tf
        self.timg = tf.Variable(np.zeros(shape), dtype=tf.float32)
        # self.tlab = tf.Variable(np.zeros((batch_size,num_labels)), dtype=tf.float32)
        self.const = tf.Variable(np.zeros(batch_size), dtype=tf.float32)
        self.targetimg = tf.Variable(np.zeros(target_shape), dtype=tf.float32)
        self.selfdb = tf.Variable(np.zeros(self_db_shape), dtype=tf.float32)


        # and here's what we use to assign them
        self.assign_timg = tf.placeholder(tf.float32, shape)
        # self.assign_tlab = tf.placeholder(tf.float32, (batch_size,num_labels))
        self.assign_const = tf.placeholder(tf.float32, [batch_size])
        self.assign_targetimg = tf.placeholder(tf.float32, target_shape)
        self.assign_selfdb = tf.placeholder(tf.float32, self_db_shape)
        
        # the resulting image, tanh'd to keep bounded from boxmin to boxmax
        self.boxmul = (boxmax - boxmin) / 2.
        self.boxplus = (boxmin + boxmax) / 2.
        # tf.random_uniform(shape, minval=-0.1, maxval=0.1, dtype=tf.float32)
        self.newimg = tf.tanh(modifier + self.timg) * self.boxmul + self.boxplus
        # self.origimg = tf.tanh(self.timg) * self.boxmul + self.boxplus
        self.targetimg_bounded = tf.tanh(self.targetimg) * self.boxmul + self.boxplus
        self.selfdb_bounded = tf.tanh(self.selfdb) * self.boxmul + self.boxplus
        
        
        # prediction BEFORE-SOFTMAX of the model
        self.outputNew = model.predict(self.newimg)
        # self.outputOrig = model.predict(self.origimg)
        self.outputTarg = model.predict(self.targetimg_bounded)
        self.outputSelfdb = model.predict(self.selfdb_bounded)
        
        # distance to the input data
        self.l2dist = tf.reduce_sum(tf.square(self.newimg-(tf.tanh(self.timg) * self.boxmul + self.boxplus)),[1,2,3])
        self.modifier_bounded = self.newimg-(tf.tanh(self.timg) * self.boxmul + self.boxplus)

        
        if self.TARGETED:
            loss1 = tf.reduce_sum(tf.square(self.outputNew - self.outputTarg),1)
            if self.hinge_loss:
                loss1 = loss1 - tf.reduce_sum(tf.square(self.outputNew - self.outputSelfdb),1) + self.CONFIDENCE
                loss1 = tf.maximum(loss1,0)
        else:
            loss1 = -tf.reduce_sum(tf.square(self.outputNew - self.outputTarg),1)
        # sum up the losses
        
        self.loss2 = tf.reduce_sum(self.l2dist)
        self.loss1 = tf.reduce_sum(self.const*loss1)
        # self.loss = -self.loss1+self.loss2
        self.loss = self.loss1 + self.loss2

        # Setup the adam optimizer and keep track of variables we're creating
        start_vars = set(x.name for x in tf.global_variables())
        optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE)
        self.train = optimizer.minimize(self.loss, var_list=[modifier])
        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]

        # these are the variables to initialize when we run
        self.setup = []
        self.setup.append(self.timg.assign(self.assign_timg))
        # self.setup.append(self.tlab.assign(self.assign_tlab))
        self.setup.append(self.const.assign(self.assign_const))
        self.setup.append(self.targetimg.assign(self.assign_targetimg))
        self.setup.append(self.selfdb.assign(self.assign_selfdb))
        
        self.init = tf.variables_initializer(var_list=[modifier]+new_vars)

    def attack(self, imgs, targetimg, selfdb):
        """
        Perform the L_2 attack on the given images for the given targets.

        If self.targeted is true, then the targets represents the target labels.
        If self.targeted is false, then targets are the original class labels.
        """
        r = []
        #print('go up to',len(imgs))
        for i in range(0,len(imgs),self.batch_size):
        #    print('tick',i)
            r.extend(self.attack_batch(imgs[i:i+self.batch_size], targetimg, selfdb))
        return np.array(r) #(batch_size, 32, 32, 3)

    def attack_batch(self, imgs, targetimg, selfdb):
        """
        Run the attack on a batch of images and labels.
        """
        # def compare(x,y):
        #     if not isinstance(x, (float, int, np.int64)):
        #         x = np.copy(x)
        #         if self.TARGETED:
        #             x[y] -= self.CONFIDENCE
        #         else:
        #             x[y] += self.CONFIDENCE
        #         x = np.argmax(x)
        #     if self.TARGETED:
        #         return x == y
        #     else:
        #         return x != y

        batch_size = self.batch_size

        # convert to tanh-space
        imgs = np.arctanh((imgs - self.boxplus) / self.boxmul * 0.999999)
        targetimg = np.arctanh((targetimg - self.boxplus) / self.boxmul * 0.999999)
        selfdb = np.arctanh((selfdb - self.boxplus) / self.boxmul * 0.999999)

        # set the lower and upper bounds accordingly
        # lower_bound = np.zeros(batch_size)
        CONST = np.ones(batch_size)*self.initial_const
        # upper_bound = np.ones(batch_size)*1e10

        # the best l2, score, and image attack
        # o_bestl2 = [1e10]*batch_size
        # o_bestscore = [np.zeros(128)]*batch_size
        # o_bestattack = [np.zeros(imgs[0].shape)]*batch_size # (batch_size, 32, 32, 3)
        
        # for outer_step in range(self.BINARY_SEARCH_STEPS):
        for outer_step in range(1):
            # print("O Best L2", o_bestl2)
            # completely reset adam's internal state.
            self.sess.run(self.init)
            batch = imgs[:batch_size]
            # batchlab = labs[:batch_size]
    
            # bestl2 = [1e10]*batch_size
            # bestscore = [np.zeros(128)]*batch_size

            # The last iteration (if we run many steps) repeat the search once.
            # if self.repeat == True and outer_step == self.BINARY_SEARCH_STEPS-1:
            #     CONST = upper_bound

            # set the variables so that we don't have to send them over again
            self.sess.run(self.setup, {self.assign_timg: batch,
                                       # self.assign_tlab: batchlab,
                                       self.assign_const: CONST,
                                       self.assign_targetimg: targetimg,
                                       self.assign_selfdb: selfdb})
            
            prev = 1e6

            best_loss = 99999.0
            best_nimg = np.zeros(imgs.shape)
            best_delta = np.zeros(imgs.shape)


            for iteration in range(self.MAX_ITERATIONS):
                # perform the attack 
                _, l, l2s, scores, nimg, delta = self.sess.run([self.train, self.loss, 
                                                         self.l2dist, self.outputNew, 
                                                         self.newimg, self.modifier_bounded])

                # print out the losses every 10%
                if iteration%(self.MAX_ITERATIONS//10) == 0:
                    print(iteration,self.sess.run((self.loss,self.loss1,self.loss2)))
                    # print(scores)

                # check if we should abort search if we're getting nowhere.
                if self.ABORT_EARLY and iteration%(self.MAX_ITERATIONS//10) == 0:
                    if l > prev*.9999:
                        break
                    prev = l


                for e,(cur_l,cur_nimg,cur_delta) in enumerate(zip(np.asarray([l]),nimg,delta)):
                    if(cur_l < best_loss):
                        best_nimg[e] = cur_nimg
                        best_delta[e] = cur_delta

                # adjust the best result found so far
                # e in range(batch_size)
            #     for e,(l2,sc,ii) in enumerate(zip(l2s,scores,nimg)):
            #         # if l2 < bestl2[e] and compare(sc, np.argmax(batchlab[e])):
            #         if l2 < bestl2[e]:
            #             bestl2[e] = l2
            #             bestscore[e] = sc
            #         # if l2 < o_bestl2[e] and compare(sc, np.argmax(batchlab[e])):
            #         if l2 < o_bestl2[e]
            #             o_bestl2[e] = l2
            #             o_bestscore[e] = sc
            #             o_bestattack[e] = ii

            # # adjust the constant as needed
            # for e in range(batch_size):
            #     if compare(bestscore[e], np.argmax(batchlab[e])) and bestscore[e] != -1:
            #         # success, divide const by two
            #         upper_bound[e] = min(upper_bound[e],CONST[e])
            #         if upper_bound[e] < 1e9:
            #             CONST[e] = (lower_bound[e] + upper_bound[e])/2
            #     else:
            #         # failure, either multiply by 10 if no solution found yet
            #         #          or do binary search with the known upper bound
            #         lower_bound[e] = max(lower_bound[e],CONST[e])
            #         if upper_bound[e] < 1e9:
            #             CONST[e] = (lower_bound[e] + upper_bound[e])/2
            #         else:
            #             CONST[e] *= 10

        # # return the best solution found
        # o_bestl2 = np.array(o_bestl2)
        # return nimg, delta
        return best_nimg, best_delta
